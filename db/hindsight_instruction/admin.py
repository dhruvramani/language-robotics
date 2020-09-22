from django.contrib import admin
from .models import SurrealRoboticsSuiteInstruction, USCFurnitureInstruction, RLBenchInstruction

admin.site.register(RLBenchInstruction)
admin.site.register(SurrealRoboticsSuiteInstruction)
admin.site.register(USCFurnitureInstruction)
