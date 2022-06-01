picom-trans --toggle --current
scrot --select --freeze -e 'mv $f /tmp/screenshot.png' 
cp /tmp/screenshot.png ~/screenshots/(date --iso-8601=seconds).png 
xclip -selection clipboard -target image/png -i /tmp/screenshot.png
picom-trans --toggle --current
