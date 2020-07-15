from django import forms

class InstructionForm(forms.Form):
    episode_id = models.CharField(max_length=36)
    instruction = models.CharField(max_length=int(1e4))