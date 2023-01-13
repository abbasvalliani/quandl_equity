import csv
import sys

def create_ddl(file):
    ddl = ''
    index = ''
    dict = load_table_definition(file)
    for table in dict:
        res = create_ddl_and_index(table, dict)
        ddl = ddl + res['ddl'] + "\n"
        index = index + res['index'] + "\n"
    return {'ddl': ddl, 'index' : index}


def load_table_definition(file):
    ret = {}
    input_file = csv.DictReader(open(file))

    for row in input_file:
        table = row["table"]
        indicator = row["indicator"]
        isfilter = row["isfilter"]
        isprimarykey = row["isprimarykey"]
        title = row["title"]
        description = row["description"]
        unittype = row["unittype"]
        if table == 'table':
            table = table + "_1"
        if indicator == 'table':
            indicator = indicator + "_1"
        item = {'indicator': indicator,
                'isfilter': isfilter,
                'isprimarykey': isprimarykey,
                'title': title,
                'description': description,
                'unittype': unittype}
        if table not in ret:
            ret[table] = []
        ret[table].append(item)
    return ret

def get_column_map(table_name, column_name, lookup):
    if table_name == 'ACTIONS' and column_name == 'companysite':
        return 'varchar(1000)'
    if table_name == 'INDICATORS' and column_name == 'description':
        return 'varchar(3000)'
    map = {'currency': 'bigint',
           'currency/share': 'decimal(14,3)',
           'date (YYYY-MM-DD)': 'date',
           'N/A': 'varchar(255)',
           'numeric': 'bigint',
           'percent': 'decimal(11,6)',
           '%': 'decimal(11,6)',
           'ratio': 'decimal(14,4)',
           'text': 'varchar(255)',
           'units': 'bigint',
           'USD': 'double',
           'USD millions': 'double',
           'USD/share': 'decimal(16,4)',
           'Y/N': 'char(1)'
           }
    if lookup not in map:
        raise Exception("cannot find lookup for column type " + lookup)
    return map[lookup]


def create_ddl_and_index(table, definition):
    ddl = 'DROP TABLE IF EXISTS ' + table + ";" + "\n"
    ddl = ddl + 'CREATE TABLE IF NOT EXISTS ' + table + "(" + "\n"
    index = ''
    if table not in definition:
        raise Exception("cannot find table definition for " + table)
    fields = definition[table]
    length = len(fields)
    count = 0

    # add the primarykeys
    primary_key = []

    for field in fields:
        count = count + 1
        column_name = field['indicator']
        column_type = field['unittype']
        is_index = field['isfilter']
        is_pk = field['isprimarykey']
        row = "\t" + column_name + "\t" + get_column_map(table, column_name, column_type)
        if is_pk == 'Y':
            primary_key.append(column_name)
        if count != length:
            row = row + ","
        row = row + "\n"
        ddl = ddl + row
        if is_pk == 'N' and is_index == 'Y':
            index_name = 'idx_' + table + "_" + column_name
            index = index + 'CREATE INDEX %s ON %s(%s)' % (index_name, table, column_name) + ";\n"
    ddl = ddl + ");"

    pk_string = ''
    pk_length = len(primary_key)
    pk_count = 0
    for pk in primary_key:
        pk_count = pk_count + 1
        pk_string = pk_string + pk
        if pk_count != pk_length:
            pk_string = pk_string + ","
    if pk_count > 0:
        index_name = 'pk_' + table
        index = index + \
                'ALTER TABLE %s ADD CONSTRAINT %s PRIMARY KEY (%s)' % (table, index_name, pk_string) + ";\n"
    return {'ddl' :ddl, 'index': index}


if __name__ == "__main__":
    file = sys.argv[1]
    ret = create_ddl(file)

    print('DDL:' + ret["ddl"]);
    print('INDEX:' + ret["index"]);
