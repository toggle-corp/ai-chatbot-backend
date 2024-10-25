#!/bin/bash -x

python manage.py collectstatic --noinput &
python manage.py migrate --noinput

gunicorn main.wsgi:application --bind 0.0.0.0:80
# gunicorn main.asgi:application --bind 0.0.0.0:80
