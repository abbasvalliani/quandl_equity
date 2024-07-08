# We are importing the cron from all /etc/cron.d/crons

printenv > /etc/environment

crontab /etc/cron.d/crontab

# Starting the cron daemon. The & is important and will allow to execute scripts below
cron -f &

# You can execute any other script below this line
while ! mysqladmin ping -h"$MYSQL_HOST" --silent; do
    echo "Waiting for $MYSQL_HOST to come up"
    sleep 1
done
echo "$MYSQL_HOST is up now."
bash /var/scripts/equity_download.sh >> /var/log/import.log 2>&1

# We'll tail logs
tail -f /var/log/import.log
