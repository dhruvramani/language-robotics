# Generated by Django 3.1 on 2020-09-26 15:34

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('hindsight_instruction', '0003_auto_20200922_0658'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='instruction',
            name='user',
        ),
    ]
