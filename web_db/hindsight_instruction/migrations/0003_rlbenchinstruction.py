# Generated by Django 3.1 on 2020-09-09 12:40

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('hindsight_instruction', '0002_auto_20200826_0804'),
    ]

    operations = [
        migrations.CreateModel(
            name='RLBenchInstruction',
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
