function generate_low_resolution_video(inputVideoPath, outputVideoPath, targetResolution)
    % PROCESS_LOW_RES_VIDEO Downsamples HD videos (landscape or portrait)
    % while maintaining orientation and correct aspect ratio.

    if ~exist(inputVideoPath, 'file')
        error('Input video file not found: %s', inputVideoPath);
    end
    inputReader = VideoReader(inputVideoPath);

    % Detect if input video is vertical (portrait)
    isVertical = inputReader.Height > inputReader.Width;

    % Assign target dimensions based on orientation
    switch upper(targetResolution)
        case 'VGA'
            if isVertical
                targetDims = [640, 480];  % [Height, Width] for Portrait VGA
            else
                targetDims = [480, 640];  % [Height, Width] for Landscape VGA
            end
        case 'QVGA'
            if isVertical
                targetDims = [320, 240];  % [Height, Width] for Portrait QVGA
            else
                targetDims = [240, 320];  % [Height, Width] for Landscape QVGA
            end
        otherwise
            error('Invalid target resolution. Choose either ''VGA'' or ''QVGA''.');
    end

    outputWriter = VideoWriter(outputVideoPath, 'MPEG-4');
    outputWriter.FrameRate = inputReader.FrameRate;
    open(outputWriter);

    fprintf('Processing: %s (%dx%d) -> Target: %s (%dx%d)...\n', ...
        inputVideoPath, inputReader.Width, inputReader.Height, ...
        upper(targetResolution), targetDims(2), targetDims(1));

    frameCount = 0;
    while hasFrame(inputReader)
        hdFrame = readFrame(inputReader);
        
        % Downsample using bicubic interpolation
        lowResFrame = imresize(hdFrame, targetDims, 'bicubic');
        
        writeVideo(outputWriter, lowResFrame);
        frameCount = frameCount + 1;
    end

    close(outputWriter);
    fprintf('Successfully processed %d frames and saved to %s.\n', frameCount, outputVideoPath);
end