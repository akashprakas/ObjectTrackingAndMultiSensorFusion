# ObjectTrackingAndMultiSensorFusion

This is a conversion of a Matlab repo for [MultiSensorFusion](https://github.com/UditBhaskar19/OBJECT_TRACKING_MULTI_SENSOR_FUSION) from Udit Bhaskar. In that repo you can find a detailed write of the setup and if you have matlab it will be better to follow that repo. I will anyway write up a small introduction  for starting up but if you want a detailed info please go through this [pdf](https://github.com/UditBhaskar19/OBJECT_TRACKING_MULTI_SENSOR_FUSION/blob/main/Sensor_Fusion_Project_version1.pdf).

The main simulation can be run with and you can put breakpoints and step through the code.

`python main.py`

## Setting up the DataBus

1. Here we read in the clusters from using 6 radar and 8 cameras in the setup. Intially we set up the intial internal and extinsic calibration paramers for both the sensors.
2. Here I am looping for 150 time stamps where the differnce between two timestamps is 0.05 seconds.
3. To read in data from each time stamp we have a radar sensor interface where we read in the senor data for the 6 radar and 8 cameras for the corresponding time stamp. The ego motion data for the corresponding timestamp is also read in.
4. Here its assumed that the maximum number of detections per sensor is 200 , so for the 6 radars the measurement matrix size will  6x200 where each element of a the matrix will be an object holding different info like below data class and for the video sensor it will be 8x200 matrix.

```python
@dataclass
class CRadarCanBus:

    measID: int = 0
    sensorID: int = 0
    detTimeStamp: float = 0.0
    px: float = 0.0
    py: float = 0.0
    measNoise: np.array = noise
    snr: float = 0.0

@dataclass
class CcameraCanBus:

    measID: int = 0
    sensorID: int = 0
    detTimeStamp: float = 0.0
    px: float = 0.0
    py: float = 0.0
    measNoise: np.array = noise
    objClassID: int = 0

````

## Coordinate transformation to the ego frame Coordinate

1. The coordinates are transformed from the sensor coordinate to the ego coordinate using the read in data and the sensor extrinsic parameters.
2. This is done for both the camera and radar .

## Radar Clustering

1. There are mainly two function here , the first one will use DBSCAN and  find the number of clusters from all the data points,
2. The second function will take the segragate the contribution from each sensor which will be later used for weighting.

## State Prediction of Radar Tracks

1. Here ego motion compensation is done as first step
2. Then the tracks from the previous cycle are predicted to the current timestamp using a constant velocity model.

## Radar Measurement Cluster Gating and Identify ungated Clusters

1. First function is returining when there is no tracks . If there are valid tracks then we loop through each tracks and each track is associated with each measurements and the log-likehood is calculated.
2. Second function find the indexes of all the gated measurements and pass it to the next function which calculated the ungated indexes.
3. In the very initial cycle all the measuremnts will be ungated or when we get a new object that will also be ungated, so the third function counts the number of unassociated clusters and return that.

## Radar Sensor Fusion

1. When there is no tracks these functions return without doing anyting.
If there are valid tracks for each track we loop through them and then loop the assignment matrix , and if the track was having a valid association an update is done in the  2D measurement space, that is we are only using the measurements px and py for updating the track. Thus the track is updated with the measurement from each 6 radar sensor if there is a valid assignement.
2. Now we have a tracks fused with each radar sensors, but we want one single track for a specific object  from radar side.
3. So  we Approximate a Gaussian mixture density as a single Gaussian using moment matching. After that we get a single track for a specific object.

## Radar Track Management

1. The init new track will initialize new tracks based on the unassociated clusters, this will assign a new id to each unassiated cluster/object.
2. Maintain existing track will mainting the tracks like how much time it has been tracked, is it lost or not .
3. Finally the delete track will delete the track list if the tracker is lost.

## Track Management for the Camera Part

1. The track management remains essentially the same following through the same step as that of the radar part. Please read through the code to understand more.

## Track To Track Fusion for Radar and Camera Tracks

1. The fusion trackers are maintained as separted track list. Initially those trackers are predicted to the current time stamp using aconstant velocity model.
2. In the gating part we loop through each track and for each track a loop is there is for a loop for radar and separate loop for camera also where we associate the previous fused track with the current sensor measurements using mehanabolis distance.
3. Now in the heterogenous fusion function we loop through each fused track , find the corressponing camera and radar track and fuse them using `Covariance Intersection`.
4. If there are ungated fused tracks(it can either be the first cycle or a new object have been introduced into the scene). Here initially we initially loop through the camera track , then finds the associated radar track to it by taking the statistical distance between them . Then take a weighted average of them and make it as a new track. If radar tracks are remaining after this they are also added to the fused track list.
5. Then the reaming ones are for maintining the existing tracks and works as explained in the radar session.

All the above write is pretty brief and missing on details, please put breakpoint and go through the code to understand it more.
Note : The code was written in a very short span to time and is having many errors.
