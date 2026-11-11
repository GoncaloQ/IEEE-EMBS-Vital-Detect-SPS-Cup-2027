function generate_partial_occlusion_video(inputVideoPath, outputVideoPath)
%generates a video of the occluding the bottom half of the face from an
%input face video
% input and output videos should both be in .mp4 format

    %initialize video Reader and writer
    videoReader = VideoReader(inputVideoPath);
    videoWriter = VideoWriter(outputVideoPath, 'MPEG-4');
    videoWriter.FrameRate = videoReader.FrameRate;
    open(videoWriter);
    
    %initialize face detector and point tracker
    faceDetector = vision.CascadeObjectDetector('MinSize', [80 80]);
    pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
    
    trackingInitialized = false;
    currentRectPoly = [];
    oldPoints = [];
    maskColor = [0, 0, 0]; %black
    
    %sale memory to lock true face dimensions across the entire video
    lockedWidth = 0;
    lockedHeight = 0;
    
    while hasFrame(videoReader)
        frame = readFrame(videoReader);
        grayFrame = rgb2gray(frame);
        
        if ~trackingInitialized
            %detect face bounding box
            bboxes = step(faceDetector, grayFrame);
            
            if ~isempty(bboxes)
                %detect the biggest face bounding box
                [~, maxIdx] = max(bboxes(:, 3) .* bboxes(:, 4));
                faceBox = double(bboxes(maxIdx, :));
                
                fx = faceBox(1);
                fy = faceBox(2);
                fw = faceBox(3); 
                fh = faceBox(4);
                
                %lock true face dimensions on first valid detection
                if lockedWidth == 0
                    lockedWidth = fw;
                    lockedHeight = fh;
                else
                    %use the maximum observed full-frontal face scale
                    lockedWidth = max(lockedWidth, fw);
                    lockedHeight = max(lockedHeight, fh);
                end
                
                % LOCKED UPRIGHT LOWER-FACE RECTANGLE OF LOWER FACE
                padX = 0.15 * lockedWidth; %padding to the sides of the face (we can adjust this)
                
                rectX = fx - padX;
                rectY = fy + (0.50 * lockedHeight);
                rectW = lockedWidth + (2 * padX);
                rectH = 0.6 * lockedHeight; %extends past chin onto upper neck
                
                %horizontal rectangle: [top-left; top-right; bottom-right; bottom-left]
                initialRectPoly = [rectX,rectY; rectX + rectW, rectY;rectX + rectW, rectY + rectH; rectX, rectY + rectH];
                
                %extract feature points across upper face region
                %this part of the face is not covered, therefore we can
                %track it's movement
                upperFaceROI = [round(fx), round(fy), round(fw), round(0.60 * fh)];
                
                %oundary safety check for image frame
                %in case the face is on the edge of the video
                upperFaceROI(1) = max(1, upperFaceROI(1));
                upperFaceROI(2) = max(1, upperFaceROI(2));
                upperFaceROI(3) = min(size(frame, 2) - upperFaceROI(1), upperFaceROI(3));
                upperFaceROI(4) = min(size(frame, 1) - upperFaceROI(2), upperFaceROI(4));
                
                %choose upper face landmarks to track
                points = detectMinEigenFeatures(grayFrame, 'ROI', upperFaceROI);
                oldPoints = points.Location;                
                if size(oldPoints, 1) >= 10 %track at least 10 face landmarks (stability)
                    initialize(pointTracker, oldPoints, grayFrame);
                    trackingInitialized = true;
                    currentRectPoly = initialRectPoly;
                end
            end
        else

            % track face landmarks across frames
            [trackedPoints, isFound] = step(pointTracker, grayFrame);
            visiblePoints = trackedPoints(isFound, :);
            oldInliers = oldPoints(isFound, :);
            
            if size(visiblePoints, 1) >= 8
                % determine horizontal translation of the points
                % prevents any diagonal rotation or shrinking when feature points compress during head turns
                [xform, inlierIdx] = estimateGeometricTransform2D(oldInliers, visiblePoints, 'translation', 'MaxDistance', 4);
                
                %translate mask rectangle position
                currentRectPoly = transformPointsForward(xform, currentRectPoly);
                
                %update tracking memory
                visiblePoints = visiblePoints(inlierIdx, :);
                setPoints(pointTracker, visiblePoints);
                oldPoints = visiblePoints;
            else
                %reset tracker if features drop below threshold
                trackingInitialized = false;
                release(pointTracker);
            end
        end
        
        %render mask
        if trackingInitialized && ~isempty(currentRectPoly)
            polyPoints = reshape(currentRectPoly', 1, []);
            frame = insertShape(frame, 'FilledPolygon', polyPoints, 'Color', maskColor, 'Opacity', 0.95);
        end
        
        %overlay of the mask and the video
        writeVideo(videoWriter, frame);
    end
    
    release(pointTracker);
    release(faceDetector);
    close(videoWriter);
    disp(['Masked video generated successfully: ', outputVideoPath]);
end

% Usage Example
%inputVideo  = 'input_video.mp4';
%outputVideo = 'masked_face_video.mp4';
%generate_partial_occlusion_video(inputVideo, outputVideo);