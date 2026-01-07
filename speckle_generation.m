%% Speckle Imaging Simulation
%
% Description: 
%   Simulates the optical propagation of an object through a scattering medium 
%   (modeled by a Phase Mask) to the detector (CCD).
%   It uses Pseudo-thermal light averaging (N iterations) to simulate the 
%   spatial coherence properties.
%
% Author: Xuyu Zhang

close all; clear; clc;
tic; % Start timer

% --- 0. Environment Setup ---
% Attempt to reset the GPU to free up memory from previous runs
try
    gpuDevice(1); 
catch
    warning('No GPU device found or failed to reset.');
end
parallel.gpu.enableCUDAForwardCompatibility(true); 

% Path Configuration
% Note: Use relative paths or modify these for your specific directory structure.
base_path = './dataset/'; 
input_pattern_path = fullfile(base_path, 'object', '5.bmp'); % Input object image
phasemask_dir = fullfile(base_path, 'phasemask/');           % Directory containing phase mask .mat files
output_dir = fullfile(base_path, 'output_patterns/');        % Output directory

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% --- 1. Physical Parameters ---
lambda = single(0.532e-6);   % Wavelength (meters) - Single precision
z1 = 0.15;                   % Distance: Source -> Object
z2 = 0.20;                   % Distance: Object -> Medium
z3 = 0.10;                   % Distance: Medium -> CCD (Detector)

% Grid and Size Definitions
m1 = 1000; n1 = 5.9e-6;      % Source plane (pixel count, pixel size)
m2 = 1000; n2 = 13.7e-6;     % Object plane field size
m2_obj = 64;                 % Actual object pixel size (for resizing)
m3 = 1000; n3 = 5.9e-6;      % Medium plane
m5 = 3072; n5 = 3.45e-6;     % CCD plane

% --- 2. Pre-calculation of Propagation Matrices (GPU + Single) ---
% Define an anonymous function for the Fresnel Propagator kernel
% Formula: exp(1j * pi/(lambda*z) * (x_out - x_in)^2)
calc_A = @(M_out, M_in, N_out, N_in, z) gpuArray(single(exp(1j*(pi/(lambda*z)) * ...
    ( ((1:M_out)'-round(M_out/2))*N_out - ((1:M_in)-round(M_in/2))*N_in ).^2 )));

calc_B = @(M_in, M_out, N_in, N_out, z) gpuArray(single(exp(1j*(pi/(lambda*z)) * ...
    ( ((1:M_out)-round(M_out/2))*N_out - ((1:M_in)'-round(M_in/2))*N_in ).^2 )));

fprintf('Pre-calculating propagation matrices...\n');

% Step 1: Source -> Object
A1_gpu = calc_A(m2, m1, n2, n1, z1);
B1_gpu = calc_B(m1, m2, n1, n2, z1);

% Step 2: Object -> Medium
A2_gpu = calc_A(m3, m2, n3, n2, z2);
B2_gpu = calc_B(m2, m3, n2, n3, z2);

% Step 3: Medium -> CCD
A3_gpu = calc_A(m5, m3, n5, n3, z3);
B3_gpu = calc_B(m3, m5, n3, n5, z3);

% --- 3. Prepare Static Object ---
fprintf('Preparing target object...\n');

% Initialize Light Source (Uniform plane wave assumed)
% Converted to GPU Array (Single Precision)
source_gpu = gpuArray(single(255 * ones(m1, m1))); 

% Load Object Image
if isfile(input_pattern_path)
    origin = imread(input_pattern_path);
    
    % Preprocessing: Flip to correct orientation
    origin = flipud(fliplr(origin)); 
    object_resized = double(imresize(origin, [m2_obj, m2_obj]));
    
    % Padding to match the simulation field size (m2 = 1000)
    pad_total = m2 - m2_obj;
    pad_pre = floor(pad_total / 2);
    pad_post = ceil(pad_total / 2);
    
    object_padded = padarray(object_resized, [pad_pre, pad_pre], 0, 'pre');
    object_padded = padarray(object_padded, [pad_post, pad_post], 0, 'post');
    
    % Transfer to GPU
    Object_base_gpu = gpuArray(single(object_padded));
else
    error('Input object image not found at: %s', input_pattern_path);
end

% --- 4. Main Simulation Loop ---
% List of Phase Mask indices to process
phasemask_list = [5000];
N = 3000; % Number of superpositions for pseudo-thermal light

for i = 1:length(phasemask_list)
    phasemask_name = phasemask_list(i);
    fprintf('Processing Phase Mask ID: %d ...\n', phasemask_name);
    
    % Load Phase Mask Data
    pm_path = sprintf('%s%d.mat', phasemask_dir, phasemask_name);
    if ~isfile(pm_path)
        warning('Skipping: Phase mask file not found (%s)', pm_path); 
        continue; 
    end
    
    tmp = load(pm_path);
    % Dynamic variable name handling
    fn = fieldnames(tmp);
    PHASEMASK_cpu = tmp.(fn{1});
    
    % Ensure correct dimensions
    if size(PHASEMASK_cpu, 1) ~= m3
        PHASEMASK_cpu = imresize(PHASEMASK_cpu, [m3, m3]);
    end

    % Pre-calculate the static phase term of the medium: exp(1j * phi)
    % This moves the exponential calculation outside the inner loop
    Phase_Term_gpu = gpuArray(exp(1j * single(PHASEMASK_cpu)));
    
    % Initialize Intensity Accumulator (Single precision to save VRAM)
    I_CCD_sum_gpu = gpuArray.zeros(m5, m5, 'single');
    
    % --- Inner Loop: Pseudo-thermal Averaging ---
    fprintf('  Starting %d superpositions...\n', N);
    
    for ii = 1:N
        % 1. Generate Pseudo-thermal Source Phase
        % Note: Requires external function 'Roughsurface_by_wcl'
        Height = Roughsurface_by_wcl(1.5e-3, 0.75e-6, 36e-6, 1.6e-6); 
        Height_crop = Height(501:1500, 501:1500);
        
        % Combine Source Amplitude with Random Phase
        Rand_Phase = gpuArray(single(Height_crop));
        Source_Field = source_gpu .* exp(1j * (2*pi/lambda) * Rand_Phase);
        
        % 2. Propagation: Source -> Object
        Illumination = A1_gpu * Source_Field * B1_gpu;
        Field_at_Object = Illumination .* Object_base_gpu;
        
        % 3. Propagation: Object -> Medium
        Field_before_Medium = A2_gpu * Field_at_Object * B2_gpu;
        
        % 4. Interaction with Medium
        Field_after_Medium = Field_before_Medium .* Phase_Term_gpu;
        
        % 5. Propagation: Medium -> CCD
        Field_at_CCD = A3_gpu * Field_after_Medium * B3_gpu;
        
        % 6. Accumulate Intensity
        I_CCD_sum_gpu = I_CCD_sum_gpu + abs(Field_at_CCD).^2;
        
        % Progress display
        if mod(ii, 500) == 0
            fprintf('    Progress: %d / %d\n', ii, N);
        end
    end
    
    % --- 5. Save Results ---
    % Gather from GPU and Normalize
    I_final = gather(I_CCD_sum_gpu);
    max_val = max(I_final(:));
    if max_val > 0
        I_final = I_final / max_val;
    end
    
    % Save as Bitmap Image
    filename_bmp = sprintf('speckle_%d_z2%0.2f_size%d.bmp', phasemask_name, z2, m2_obj);
    imwrite(I_final, fullfile(output_dir, filename_bmp));
    
    % Save as MAT file (Optional: contains raw float data)
    % filename_mat = sprintf('data_%d.mat', phasemask_name);
    % save(fullfile(output_dir, filename_mat), 'I_final');
    
    fprintf('Completed Phase Mask %d.\n', phasemask_name);
end

total_time = toc;
fprintf('All tasks completed. Total time: %.2f seconds.\n', total_time);