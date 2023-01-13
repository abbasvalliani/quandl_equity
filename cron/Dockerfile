FROM debian:11-slim

RUN apt-get update && apt-get install -y wget procps emacs git cron bash lsb-release \
    gnupg pip python3 && apt-get clean
RUN wget https://dev.mysql.com/get/mysql-apt-config_0.8.24-1_all.deb
RUN dpkg -i mysql-apt-config_0.8.24-1_all.deb
RUN apt-get update && apt-get install -y default-mysql-client mysql-shell  && apt-get clean

#set the environment variables in the container
ENV QUANDL_API=${QUANDL_API}
ENV MYSQL_USER=${MYSQL_USER}
ENV MYSQL_PASSWORD=${MYSQL_PASSWORD}
ENV MYSQL_HOST=${MYSQL_HOST}
ENV MYSQL_DATABASE=${MYSQL_DATABASE}

COPY python/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Adding crons from current directory
ADD ./bin/crons /etc/cron.d/crontab

# Adding helper scripts
ADD ./bin/db_connect.sh /
ADD ./bin/logs.sh /

# Adding scripts from current directory
RUN mkdir -p /var/scripts
COPY ./bin/*.sh ./python/*.py /var/scripts

# Adding executable permissions
RUN ls -la /var/scripts
RUN chmod +x /*.sh /var/scripts /var/scripts/*.sh /var/scripts/*.py /etc/cron.d/crontab

ENTRYPOINT /var/scripts/entrypoint.sh
