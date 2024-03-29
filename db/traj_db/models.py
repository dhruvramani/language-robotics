import uuid
from django.db import models
from django.conf import settings
from polymorphic.models import PolymorphicModel

class TrajectoryTag(models.Model):
    '''Table of tags that can be attached to episodes.
  
        + Attributes:
            - name: Human readable tag name.

        object.trajectory_set : The epsidoes that have been annotated with this tag.
    '''
    name = models.CharField(max_length=75)
    
    def __str__(self):
        return self.name

class Trajectory(PolymorphicModel):
    ''' Abstract table desicribing trajectory schema.
        NOTE : ABSTRACT CLASS - Create a new subclass/table for each Env.
        NOTE : EACH environment has it's own seperate table to store data.

        REFER: https://django-polymorphic.readthedocs.io/en/latest/quickstart.html
               https://stackoverflow.com/questions/30343212/foreignkey-field-related-to-abstract-model-in-django 
        
        + Attributes:
            - episode_id: A UUID to identify the trajectory.
            - traj_count: The index number to this trajectory.
            - env_id: An identifier for the environment. Usually, the environment name.
            - task_id: Stores config.env_type - corresponding to the behavior in the trajectory.
            - traj_steps: Number of time-steps in the trajectory.
            - data_path: Filename holding the trajectory data for this episode.
            - is_archived: Wether the trajectory is present in a tar.gz file or not
            - time_stamp: Unix timestamp recording when the trajectory was generated.
            - episode_type: The type of policy that generated the episode.  Possible values are:
                > `EPISODE_ROBOT_IMITATED`: Generated by a script which imitated human-generated play data.
                > `EPISODE_ROBOT_PLAY`: Generated from teleoperated human demonstration as play data.
                > `EPISODE_ROBOT_EXPERT` : Generated from expert human demonstration to solve a particular task.
                > `EPISODE_ROBOT_POLICY`: Generated from a policy aimed to solve a particular task.
                > `EPISODE_ROBOT_EXPLORED` : Data generated from (random) exploration.
            - tags: (TrajectoryTag) A list of tags attached to this episode.
    '''
    # class Meta: # Not needed as PolymorphicModel
    #     abstract = True

    EPISODE_TYPE_CHOICES = [
        ('EPISODE_ROBOT_IMITATED', 'Script Generated by Imitation of Play-Data'),
        ('EPISODE_ROBOT_PLAY', 'Human Demonstration: Play-Data'),
        ('EPISODE_ROBOT_EXPERT', 'Human Demonstration: Expert, Task Specific'),
        ('EPISODE_ROBOT_POLICY', 'Collected by Task Specific Policy'),
        ('EPISODE_ROBOT_EXPLORED', 'Generated by (Random) Exploration'),
        ('EPISODE_ROBOT_RANDOM', 'For Testing')
    ]

    episode_id = models.UUIDField(default=uuid.uuid4, editable=False)
    traj_count = models.AutoField(primary_key=True) 
    # NOTE: In Utopia, this should be defined in subclasses. 
    #       But => both sub & super have diff. PKs - Error.
    env_id = models.CharField(max_length=50)
    task_id = models.CharField(max_length=50)
    traj_steps = models.IntegerField()
    time_stamp = models.TimeField(auto_now_add=True, auto_now=False)
    data_path = models.CharField(max_length=1024)
    is_archived = models.BooleanField(default=False)
    episode_type = models.CharField(max_length=30, choices=EPISODE_TYPE_CHOICES, default='EPISODE_ROBOT_PLAY')
    tag = models.ForeignKey(TrajectoryTag, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return "{} - {} : {}".format(self.env_id, self.task_id, self.episode_id)

class RLBenchTrajectory(Trajectory):
    ''' Trajectory table for RLBench environment. '''
    pass

class SurrealRoboticsSuiteTrajectory(Trajectory):
    ''' Trajectory table for Surreal Robotics Suite environment. '''
    pass

class USCFurnitureTrajectory(Trajectory):
    ''' Trajectory table for USC's Furniture environment. '''
    pass

class ArchiveFile(models.Model):
    ''' Table describing where episodes are stored in archives.
        This information is relevant if you want to download or extract a specific episode from the archives they are distributed in.
  
        + Attributes:
            - trajectory: Foreign key into the `Trajectory` table.
            - env_id: Which env this trajectory belongs to. 
                > Same as `trajectory.env_id`.
            - archive_file: Name of the archive file containing the corresponding episode.
    '''
    trajectory = models.ForeignKey(Trajectory, on_delete=models.CASCADE)
    env_id = models.CharField(max_length=50)
    archive_file = models.CharField(max_length=1024)

    def __str__(self):
        return "{} : {}".format(self.trajectory_.episode_id, self.archive_file)
