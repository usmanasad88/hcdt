ffmpeg -i /home/mani/Central/HaVid/S07A02I01M0.mp4 -vf fps=1 /home/mani/Central/HaVid/S07A02I01M0/frame_%04d.png

ffmpeg -i overlay-S02A08-ICL.mp4 \
  -c:v libx264 -profile:v baseline -level 3.1 -preset medium -crf 23 \
  -vf "scale='min(640,iw)':'min(480,ih)':force_original_aspect_ratio=decrease" \
  -c:a aac -b:a 128k \
  -movflags +faststart \
  overlay-S02A08-ICL-whatsapp.mp4