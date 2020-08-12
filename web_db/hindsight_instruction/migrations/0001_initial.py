# Generated by Django 3.1 on 2020-08-12 11:56

from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('contenttypes', '0002_remove_content_type_name'),
    ]

    operations = [
        migrations.CreateModel(
            name='Instruction',
            fields=[
                ('instruction_id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('instruction_count', models.IntegerField(default=1)),
                ('env_id', models.CharField(max_length=50)),
                ('task_id', models.CharField(max_length=50)),
                ('instruction', models.CharField(max_length=10000)),
                ('polymorphic_ctype', models.ForeignKey(editable=False, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='polymorphic_hindsight_instruction.instruction_set+', to='contenttypes.contenttype')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
        ),
        migrations.CreateModel(
            name='SurrealRoboticsSuiteInstruction',
            fields=[
                ('instruction_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='hindsight_instruction.instruction')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('hindsight_instruction.instruction',),
        ),
        migrations.CreateModel(
            name='USCFurnitureInstruction',
            fields=[
                ('instruction_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='hindsight_instruction.instruction')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('hindsight_instruction.instruction',),
        ),
    ]
