%% main_dat_to_mat_saveH_parfor_multi.m
% Convert .dat under user1..user11 to <name>.mat (mirrored structure).
% Minimal, key English comments only.

%% ---- Config (edit as needed) ----
rootRootDir   = "C:\Users\79398\git\SSC2025\Gait\data\CSI_Gait_dat";        % parent of user1..user11
outRootDir    = "C:\Users\79398\git\SSC2025\Gait\data\CSI_Gait_mat_double"; % output parent (mirrors users)
utilsDir      = "C:\Users\79398\git\SSC2025\Gait\DataConverter\matlab_scripts\utils";                                        % contains read_bf_file.m (+ MEX)

NSC_EXPECT    = 30;        % Intel 5300: 30 subcarriers
USER_RANGE    = 1:11;      % which user folders to include
B_RANGE       = [1, 40];   % inclusive range for b
RX_SET        = [1,2,3,4,5,6]; % allowed r indices: r1..r3

USE_PARFOR    = true;      % use parfor if available
NUM_WORKERS   = [];        % [] = default pool size
CAST_TO_SINGLE = false;    % true => store CSI_mat as single to reduce file size

%% ---- Setup paths & output root ----
addpath(char(utilsDir));
if ~exist(outRootDir, 'dir'); mkdir(outRootDir); end

%% ---- Collect user folders by pattern user<ID> and filter by USER_RANGE ----
allUserDirs = dir(fullfile(rootRootDir, "user*"));
allUserDirs = allUserDirs([allUserDirs.isdir]);
users = strings(0,1);
for i = 1:numel(allUserDirs)
    m = regexp(allUserDirs(i).name, '^user(\d+)$', 'tokens', 'once');
    if ~isempty(m) && any(str2double(m{1}) == USER_RANGE)
        users(end+1,1) = string(allUserDirs(i).name); %#ok<AGROW>
    end
end
if isempty(users)
    warning("No user folders matched under %s", rootRootDir); return;
end

%% ---- Build task list (char paths; mirror output structure) ----
datPaths = {};
outPaths = {};
for u = 1:numel(users)
    userName  = char(users(u));                         % e.g., 'user3'
    inDir     = char(fullfile(rootRootDir, userName));  % input/userX
    outDir    = char(fullfile(outRootDir,  userName));  % output/userX
    if ~exist(outDir, 'dir'); mkdir(outDir); end

    f = dir(fullfile(inDir, "*.dat"));
    for k = 1:numel(f)
        fn = f(k).name;
        tok = regexp(fn, '^user(\d+)-(\d+)-(\d+)-r(\d+)\.dat$', 'tokens', 'once'); % userID-b-c-rx
        if isempty(tok); continue; end
        b  = str2double(tok{3});
        rx = str2double(tok{4});
        if isfinite(b) && isfinite(rx) && b>=B_RANGE(1) && b<=B_RANGE(end) && any(rx==RX_SET)
            datPaths{end+1,1} = char(fullfile(inDir, fn)); %#ok<AGROW>
            outPaths{end+1,1} = char(fullfile(outDir, [erase(fn, ".dat"), '.mat'])); %#ok<AGROW>
        end
    end
end

nTasks = numel(datPaths);
if nTasks == 0
    warning("No .dat matched: users=%s, b in [%d..%d], r in {%s}. Root: %s", ...
        mat2str(USER_RANGE), B_RANGE(1), B_RANGE(end), num2str(RX_SET), rootRootDir);
    return;
end
fprintf("Collected %d tasks across %d user folders.\n", nTasks, numel(users));

%% ---- Parallel execution (transparency-safe) ----
hasPCT = false;
if USE_PARFOR
    try
        hasPCT = license('test','Distrib_Computing_Toolbox') && ~isempty(ver('parallel'));
    catch, hasPCT = false;
    end
end

if hasPCT
    pool = gcp('nocreate');
    if isempty(pool)
        if isempty(NUM_WORKERS), parpool('local'); else, parpool('local', NUM_WORKERS); end
    elseif ~isempty(NUM_WORKERS) && pool.NumWorkers ~= NUM_WORKERS
        delete(pool); parpool('local', NUM_WORKERS);
    end
    % Ensure workers see utilsDir (and its subfolders)
    pctRunOnAll(sprintf('addpath(''%s'');', char(genpath(char(utilsDir)))));

    parfor t = 1:nTasks
        dp = datPaths{t}; op = outPaths{t}; %#ok<PFBNS> % local copies
        try
            process_one(dp, op, NSC_EXPECT, CAST_TO_SINGLE);
        catch ME
            warning("Failed on %s (%s)", dp, ME.message);
        end
    end
else
    for t = 1:nTasks
        dp = datPaths{t}; op = outPaths{t};
        try
            process_one(dp, op, NSC_EXPECT, CAST_TO_SINGLE);
        catch ME
            warning("Failed on %s (%s)", dp, ME.message);
        end
    end
end

fprintf("All done! Saved .mat files under %s (mirrored users).\n", outRootDir);

%% ================= Worker-safe function =================
function process_one(datPath, outPath, NSC_EXPECT, CAST_TO_SINGLE)
    fprintf("Processing %s …\n", datPath);

    % ensure output subdir exists
    p = fileparts(outPath);
    if ~exist(p, 'dir'); mkdir(p); end

    % 1) read raw frames (Intel 5300 toolbox return: cell of structs)
    CSI_cell = read_bf_file(datPath);
    if isempty(CSI_cell)
        warning("%s contains no readable frames – skipped", datPath); return;
    end

    % 2) extract CSI_mat (T x NSC_EXPECT x 3); prefer scaled CSI if available
    [CSI_mat, ~] = extract_csi_tensor(CSI_cell, NSC_EXPECT);
    if CAST_TO_SINGLE, CSI_mat = single(CSI_mat); end

    % 3) save compressed (-v7), fallback to -v7.3 on failure (e.g., >2GB)
    try
        save(outPath, 'CSI_mat', '-v7');
    catch
        save(outPath, 'CSI_mat', '-v7.3');
    end

    fprintf("Saved %s (CSI_mat: %dx%dx%d, %s)\n", ...
        outPath, size(CSI_mat,1), size(CSI_mat,2), size(CSI_mat,3), class(CSI_mat));
end

%% ================= Helper (self-contained) =================
function [H, rssi] = extract_csi_tensor(cellArr, NSC)
% Keep frames with exactly NSC subcarriers; bring SC to dim-1; keep 3 Rx (pad NaN if <3).
    T = numel(cellArr);
    list = cell(T,1);
    ra = nan(T,1); rb = ra; rc = ra;
    keep = false(T,1);

    for i = 1:T
        e = cellArr{i};
        csi = [];
        try
            csi = get_scaled_csi(e);   % use scaled CSI if toolbox provides it
        catch
            if isstruct(e) && isfield(e, "csi"), csi = e.csi; end
        end
        if isempty(csi), continue; end

        sz  = size(csi);
        idx = find(sz == NSC, 1);
        if isempty(idx), continue; end

        if idx ~= 1
            perm = 1:numel(sz); perm([1, idx]) = [idx, 1];
            csi = permute(csi, perm);
        end

        csi = reshape(csi, NSC, []);      % (NSC x Nrx)
        if size(csi,2) < 3, csi(:, 2:3) = nan; end
        list{i} = reshape(csi(:,1:3), [1 NSC 3]);

        if isfield(e, "rssi_a"), ra(i) = double(e.rssi_a); end
        if isfield(e, "rssi_b"), rb(i) = double(e.rssi_b); end
        if isfield(e, "rssi_c"), rc(i) = double(e.rssi_c); end

        keep(i) = true;
    end

    if ~any(keep), error("No frames with %d sub-carriers detected", NSC); end

    H = cat(1, list{keep});           % T x NSC x 3
    rssi.a = ra(keep); rssi.b = rb(keep); rssi.c = rc(keep);
end
