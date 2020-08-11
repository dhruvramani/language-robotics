import sys
import uuid
from django.db import models
from django.conf import settings
from polymorphic.models import PolymorphicModel

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from traj_db.models import Trajectory

class Instruction(PolymorphicModel):
    ''' Table containing instructios for each trajectory collected using Hindsight Instruction.
        NOTE : ABSTRACT CLASS - Create a new subclass/table for each Env.

        + Attributes:
            - instruction_id: A UUID providing a uique identifier for the instruction.
            - instruction_count: The index number to this instruction.
            - env_id: An identifier for the environment. Same as trajectory.
            - task_id: Stores config.env_type. Same  as trajectory.
            - user: The user who annotated the trajectory. 
                > NOTE : Set it to `request.user`.
            - instruction: collected instruction for a particular trajectory.
            - trajectory: (Trajectory) trajectory associated with the instruction.
                > Chosen this way as a single trajectoy can have multiple instructions.
                > Also while training, we only need trajectoies containing instructions. Easier this way.
    ''' 
    instruction_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    instruction_count = models.AutoField()
    env_id = models.CharField(max_length=50)
    task_id = models.CharField(max_length=50)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    instruction = models.CharField(max_length=int(1e4))
    trajectory = models.ForeignKey(Trajectory, on_delete=models.CASCADE)

    def __str__(self):
        return "{} : {}".format(env_id, instruction)

class SurrealRoboticsSuiteInstruction(Instruction):
    ''' Instruction table for Surreal Robotics Suite environment. '''
    pass

class USCFurnitureInstruction(Instruction):
    '''Instruction table for USC's Furniture environment. '''
    pass
