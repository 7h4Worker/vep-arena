clear;
clc;

subject = getenv('HD_SUBJECT');
if isempty(subject)
    subject = 'S1';
end
dataLengthText = getenv('HD_WINDOW_MS');
if isempty(dataLengthText)
    dataLength = 500;
else
    dataLength = str2double(dataLengthText);
end

datasetRoot = getenv('HD_DATASET_ROOT');
if isempty(datasetRoot)
    datasetRoot = 'D:/ProjData/datasets/ssvep_hd_200target';
end
projectRoot = getenv('HD_PROJECT_ROOT');
if isempty(projectRoot)
    projectRoot = 'D:/ProjData/proj_python/vep_arena';
end

codeRoot = fullfile(datasetRoot, 'raw', 'code_data', 'code&data');
cacheRoot = fullfile(datasetRoot, 'derivatives', 'tdca_sample');
resultRoot = fullfile(projectRoot, 'tasks', 'ssvep_hd_200target_tdca_sample', 'results');
if ~exist(resultRoot, 'dir')
    mkdir(resultRoot);
end
addpath(fullfile(codeRoot, 'TDCA_function'));

h5File = fullfile(cacheRoot, sprintf('filtered_%s_200target_66ch_18blocks.h5', subject));
outCsv = fullfile(resultRoot, sprintf('tdca_%s_200target_66ch_18blocks_w%d_matlab_cache.csv', subject, dataLength));
outMat = fullfile(resultRoot, sprintf('tdca_%s_200target_66ch_18blocks_w%d_matlab_cache.mat', subject, dataLength));

latency = round(140 / 4);
lag = 5;
targetNum = 200;
fbNum = 5;
neededSamples = dataLength / 4 + latency + lag;
if neededSamples > 185
    fprintf('subject=%s window_ms=%d unavailable requires_samples=%.0f file_samples=185\n', subject, dataLength, neededSamples);
    fid = fopen(outCsv, 'w');
    fprintf(fid, 'subject,targets,channels,filter_banks,blocks,window_ms,accuracy,itr_bpm,seconds,status,reason\n');
    fprintf(fid, '%s,200,66,5,18,%d,,,,unavailable,requires %.0f samples but offline file has 185\n', subject, dataLength, neededSamples);
    fclose(fid);
    return;
end

fprintf('Reading %s\n', h5File);
bpdatahAll = h5read(h5File, '/bpdatahAll');
bpdatahAll = permute(bpdatahAll, [5 4 3 2 1]);
bpdatahAll = single(bpdatahAll);
fprintf('bpdatahAll size: ');
fprintf('%d ', size(bpdatahAll));
fprintf('\n');

freqs = reshape(repmat([8:1:15, 8.2:1:15.2, 8.4:1:15.4, 8.6:1:15.6, 8.8:1:15.8], 5, 1), 200, 1);

tic;
accAll = sub_BCIAnalysis_hs_single(bpdatahAll, freqs, latency, dataLength, lag);
seconds = toc;

accuracy = accAll(fbNum);
itr_bpm = ITR(targetNum, accuracy, 60 / (dataLength / 1000 + 0.5));

fprintf('subject=%s targets=200 channels=66 blocks=18 window_ms=%d accuracy=%.12f itr_bpm=%.12f seconds=%.3f\n', subject, dataLength, accuracy, itr_bpm, seconds);

fid = fopen(outCsv, 'w');
fprintf(fid, 'subject,targets,channels,filter_banks,blocks,window_ms,accuracy,itr_bpm,seconds,status,reason\n');
fprintf(fid, '%s,200,66,5,18,%d,%.15g,%.15g,%.15g,complete,\n', subject, dataLength, accuracy, itr_bpm, seconds);
fclose(fid);

save(outMat, 'accAll', 'accuracy', 'itr_bpm', 'seconds', 'dataLength', 'targetNum', 'subject');
fprintf('Wrote %s\n', outCsv);
fprintf('Wrote %s\n', outMat);

function ACCU = sub_BCIAnalysis_hs_single(datum9, freqV, latency, time, lag)
    [n_channel, ~, condition, n_block, FBNum] = size(datum9);
    subspace = n_channel;
    acc_tdca = zeros(FBNum, length(time), n_block);
    for cv = 1:n_block
        train_ind = setdiff(1:n_block, cv);
        datatrain = datum9(:, :, :, train_ind, :);
        datatest = squeeze(datum9(:, :, :, cv, :));
        for k = 1:length(time)
            fprintf('cv=%d/%d window=%d\n', cv, n_block, time(k));
            model = tdca_train_wofb_hs_single(datatrain, time(k) / 4, freqV, latency, lag);
            cnt = zeros(1, FBNum);
            for ii = 1:condition
                epoch = squeeze(datatest(:, :, ii, :));
                prediction = tdca_test_wofb_hs(epoch, condition, model, time(k) / 4, latency, lag, subspace);
                for nfb = 1:FBNum
                    cnt(nfb) = cnt(nfb) + (ii == prediction(nfb));
                end
            end
            acc_tdca(:, k, cv) = cnt / condition;
            fprintf('  fb5_acc_so_far=%.6f\n', mean(squeeze(acc_tdca(FBNum, k, 1:cv))));
        end
    end
    ACCU = mean(acc_tdca, 3);
end

function model = tdca_train_wofb_hs_single(data, time, freq, latency, lag)
    [n_chan, ~, n_cond, n_block, n_band] = size(data);
    bpdata = data(:, 1:time + latency + lag, :, :, :);
    n_dim = lag * n_chan;
    traindatah = zeros(n_dim, time * 2, n_cond, n_band, 'single');
    SFshf = zeros(n_dim, n_dim, n_band);
    P = cell(1, n_cond, n_band);

    for fbi = 1:n_band
        P_f = projection_matrix_mem_single(time, 250, freq);
        P(:, :, fbi) = P_f;

        bpdatah = zeros(n_dim, time, n_cond, n_block, 'single');
        for l = 1:lag
            rows = (l - 1) * n_chan + (1:n_chan);
            bpdatah(rows, :, :, :) = bpdata(:, latency + l:time + latency + l - 1, :, :, fbi);
        end

        bpdatahp = zeros(n_dim, time * 2, n_cond, n_block, 'single');
        template_sum = zeros(n_dim, time * 2, n_cond, 'single');
        for ii = 1:n_block
            for cond = 1:n_cond
                P_cond = P_f{cond};
                trial = bpdatah(:, :, cond, ii);
                trial_hp = [trial, trial * P_cond'];
                bpdatahp(:, :, cond, ii) = trial_hp;
                template_sum(:, :, cond) = template_sum(:, :, cond) + trial_hp;
            end
        end

        traindatah(:, :, :, fbi) = template_sum / n_block;
        trca_Xm = traindatah(:, :, :, fbi);
        trca_Xm = trca_Xm - mean(trca_Xm, 2);
        trca_Xma = mean(trca_Xm, 3);
        trca_Xmb = trca_Xm - repmat(trca_Xma, 1, 1, n_cond);
        Hb = double(reshape(trca_Xmb, n_dim, [])) / sqrt(n_cond);

        Sw = zeros(n_dim, n_dim);
        for cond = 1:n_cond
            X = squeeze(bpdatahp(:, :, cond, :));
            X = X - mean(X, 2);
            X = X - mean(X, 3);
            X = reshape(X, n_dim, []);
            X = double(X) / sqrt(n_block * n_cond);
            Sw = Sw + X * X';
        end

        Sb = Hb * Hb';
        [V, D] = eig(Sw \ Sb);
        [~, index] = sort(diag(D), 'descend');
        SFshf(:, :, fbi) = V(:, index);
    end

    model.traindatah = traindatah;
    model.SFsAllh = SFshf;
    model.P = P;
end

function P = projection_matrix_mem_single(time, fs, freq)
    n = (1:time) / fs;
    P = cell(1, length(freq));
    for cond = 1:length(freq)
        Y = [];
        for harmonic = 1:5
            Y = cat(2, Y, sin(2 * pi * harmonic * freq(cond) * n)', cos(2 * pi * harmonic * freq(cond) * n)');
        end
        Y = Y - repmat(mean(Y, 1), time, 1);
        [Q, ~] = qr(Y, 0);
        P{cond} = single(Q * Q');
    end
end
