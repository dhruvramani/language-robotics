## TODO 
Run :
```
pip install django-polymorphic
python manage.py migrate
python manage.py createsuperuser
python manage.py syncdb
python manage.py migrate traj_db

```
+ Each env has it's own seperate table to store data.
+ 2 Apps :
  - `traj_db` : Contains the main meta-data DB to store the trajectories. Stores the language/instruction schema too.
  - `hindsight_instruction` : Web-App to play trajectories and store instructions.

+ Might migrate to Postgres in the future. Need sqllite-3 for speed now.
  - https://www.digitalocean.com/community/tutorials/how-to-use-postgresql-with-your-django-application-on-ubuntu-14-04
  - https://www.vphventures.com/how-to-migrate-your-django-project-from-sqlite-to-postgresql/

+ To run on Google Colabs
  - https://ngrok.com/
  - https://medium.com/@kshitijvijay271199/flask-on-google-colab-f6525986797b
  - https://stackoverflow.com/questions/59741453/is-there-a-general-way-to-run-web-applications-on-google-colab
  