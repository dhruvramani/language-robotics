from django.contrib import admin
from .models import ArchiveFile, SurrealRoboticsSuiteTrajectory, USCFurnitureTrajectory, TrajectoryTag
# Register your models here.

admin.site.register(SurrealRoboticsSuiteTrajectory)
admin.site.register(USCFurnitureTrajectory)
admin.site.register(ArchiveFile)
admin.site.register(TrajectoryTag)
