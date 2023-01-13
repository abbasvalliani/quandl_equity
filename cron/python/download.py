import json
import sys
import time
import os
import shutil
import logging
from pathlib import Path

def bulk_fetch(api_key, table, parent_dir):
    Path(parent_dir).mkdir(parents=True, exist_ok=True)
    table_json_path = os.path.join(parent_dir, '%s.json' % table)
    table_json_tmp_path = os.path.join(parent_dir, '%s_temp.json' % table)
    table_zip_path = os.path.join(parent_dir, '%s.zip' % table)

    url = 'https://www.quandl.com/api/v3/datatables/SHARADAR/%s.json?qopts.export=true&api_key=%s' % (table, api_key)
    # optionally add parameters to the url to filter the data retrieved, as described in the associated table's documentation, eg here: https://www.quandl.com/databases/SF1/documentation/getting-started
    version = sys.version.split(' ')[0]
    if version < '3':
        import urllib2
        fn = urllib2.urlopen
    else:
        import urllib
        import urllib.request
        fn = urllib.request.urlopen

    valid = ['fresh', 'regenerating']
    invalid = ['generating']
    status = ''

    while status not in valid:
        logging.info('Getting meta for table %s' % (table))
        Dict = json.loads(fn(url).read())
        with open(table_json_tmp_path, 'w') as g:
            json.dump(Dict, g)
            g.close()
        last_refreshed_time = Dict['datatable_bulk_download']['datatable']['last_refreshed_time']
        status = Dict['datatable_bulk_download']['file']['status']
        link = Dict['datatable_bulk_download']['file']['link']
        logging.info('Download file status for table: %s with status: %s' % (table, status))
        if status not in valid:
            time.sleep(60)

    ret = {}
    ret['last_refreshed_time'] = last_refreshed_time

    # read the current json last update time
    if os.path.exists(table_json_path):
        lastDict = json.load(open(table_json_path))
        prior_refreshed_time = lastDict['datatable_bulk_download']['datatable']['last_refreshed_time']
        ret['prior_refreshed_time'] = prior_refreshed_time
        if prior_refreshed_time == last_refreshed_time:
            logging.info('Table: %s no updated needed for . already have recent' % (table))
            os.remove(table_json_tmp_path)
            ret['updated'] = 0
            return ret
    #else:
    #    raise Exception('Unable to download meta data for table ' + table)

    # download file
    logging.debug('Table: %s. Fetching from link %s' % (table, link))
    zipString = fn(link).read()
    f = open(table_zip_path, 'wb')
    f.write(zipString)
    f.close()

    # copy json meta
    shutil.copyfile(table_json_tmp_path, table_json_path)
    os.remove(table_json_tmp_path)

    ret['updated'] = 1
    return ret

if __name__ == "__main__":
    api_key = sys.argv[1]
    table = sys.argv[2]
    dir = sys.argv[3]
    bulk_fetch(api_key, table, dir)
    print('Done')
