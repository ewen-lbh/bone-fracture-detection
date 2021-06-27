#!/usr/bin/env fish

for f in datasets/various/*.png
    richprint $f [red] without blur
    ./edge_detection_batch.py $f 40 160 20 
    richprint $f [green] with blur
    ./edge_detection_batch.py $f 40 160 20 --blur 3
end
