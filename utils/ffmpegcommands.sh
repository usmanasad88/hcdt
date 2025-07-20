ffmpeg -i /home/mani/Central/HaVid/S07A02I01M0.mp4 -vf fps=1 /home/mani/Central/HaVid/S07A02I01M0/frame_%04d.png

ffmpeg -i cam01_icl_best.mp4 \
  -c:v libx264 -profile:v baseline -level 3.1 -preset medium -crf 23 \
  -vf "scale='min(640,iw)':'min(480,ih)':force_original_aspect_ratio=decrease" \
  -c:a aac -b:a 128k \
  -movflags +faststart \
  cam01_icl_best-whatsapp.mp4


ffmpeg -ss 00:00:00 -i cam01_icl_best.mp4 -t 00:00:08 \
-c:v libx264 \
-profile:v baseline \
-level 3.0 \
-pix_fmt yuv420p \
-crf 23 \
-preset medium \
-c:a aac \
-ac 2 \
-ar 44100 \
-b:a 128k \
-movflags +faststart \
cam01_icl_bestoutput_whatsapp.mp4

ffmpeg -i qalam.webm -vf "scale=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -preset medium -crf 23 -profile:v high -level 4.0 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart method.mp4
