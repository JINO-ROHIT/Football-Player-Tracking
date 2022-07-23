# Football-Player-Tracking ⚽

This is an object detection + tracking project problem where you have to first run an object detector over the frames and then track the detected objects over N frames.
Using a tracker reduces huge computation costs and also improves performance.

For object tracking, a simple tracker was implemented. The logic is to assign object ids according to the object detector predictions and then use a distance metric and 
depending on a threshold T, keep updating the object ids . Finally remove the ids when the objects move out of frame.


![](image.png)
