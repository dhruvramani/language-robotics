# Generated by Django 3.1 on 2020-09-22 06:41

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
            name='Trajectory',
            fields=[
                ('episode_id', models.UUIDField(default=uuid.uuid4, editable=False)),
                ('traj_count', models.AutoField(default=1, primary_key=True, serialize=False)),
                ('env_id', models.CharField(max_length=50)),
                ('task_id', models.CharField(max_length=50)),
                ('traj_steps', models.IntegerField()),
                ('time_stamp', models.TimeField(auto_now_add=True)),
                ('data_path', models.CharField(max_length=1024)),
                ('is_archived', models.BooleanField(default=False)),
                ('episode_type', models.CharField(choices=[('EPISODE_ROBOT_IMITATED', 'Script Generated by Imitation of Play-Data'), ('EPISODE_ROBOT_PLAY', 'Human Demonstration: Play-Data'), ('EPISODE_ROBOT_EXPERT', 'Human Demonstration: Expert, Task Specific'), ('EPISODE_ROBOT_POLICY', 'Collected by Task Specific Policy'), ('EPISODE_ROBOT_EXPLORED', 'Generated by (Random) Exploration'), ('EPISODE_ROBOT_RANDOM', 'For Testing')], default='EPISODE_ROBOT_PLAY', max_length=30)),
                ('polymorphic_ctype', models.ForeignKey(editable=False, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='polymorphic_traj_db.trajectory_set+', to='contenttypes.contenttype')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
        ),
        migrations.CreateModel(
            name='TrajectoryTag',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=75)),
            ],
        ),
        migrations.CreateModel(
            name='RLBenchTrajectory',
            fields=[
                ('trajectory_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='traj_db.trajectory')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('traj_db.trajectory',),
        ),
        migrations.CreateModel(
            name='SurrealRoboticsSuiteTrajectory',
            fields=[
                ('trajectory_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='traj_db.trajectory')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('traj_db.trajectory',),
        ),
        migrations.CreateModel(
            name='USCFurnitureTrajectory',
            fields=[
                ('trajectory_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='traj_db.trajectory')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('traj_db.trajectory',),
        ),
        migrations.AddField(
            model_name='trajectory',
            name='tag',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='traj_db.trajectorytag'),
        ),
        migrations.CreateModel(
            name='ArchiveFile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('env_id', models.CharField(max_length=50)),
                ('archive_file', models.CharField(max_length=1024)),
                ('trajectory', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='traj_db.trajectory')),
            ],
        ),
    ]