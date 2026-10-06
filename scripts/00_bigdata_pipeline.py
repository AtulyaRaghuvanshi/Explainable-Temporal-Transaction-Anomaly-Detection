"""Drives the Hadoop-ecosystem part of X-TAD (HDFS -> Spark -> Hive) through Docker.

Steps (run all by default, or pick some):  up  load  spark  hive  export
  up     start HDFS / Spark / Hive containers (docker compose)
  load   copy raw CSVs from data/raw into HDFS  (hdfs:///xtad/raw)
  spark  run spark/behavioral_features.py on the Spark cluster (reads+writes HDFS)
  hive   create the external Hive table over the Spark output and run sql/analytics/*.hql
  export copy the Spark feature table from HDFS to data/hdfs_export/features for the Python models
"""
import argparse, io, subprocess, sys, time
from pathlib import Path
import pandas as pd
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
from src.config import load_config

cfg = load_config(); bd = cfg['bigdata']
NN, SM, HV = bd['namenode_container'], bd['spark_container'], bd['hive_container']
HDFS = bd['hdfs_uri']
BEELINE = ['docker', 'exec', HV, 'beeline', '-u', bd['hive_jdbc'], '-n', 'hive', '--silent=true']


def run(cmd, **kw):
    print('$', ' '.join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, cwd=root, **kw)


def up():
    run(['docker', 'compose', 'up', '-d'])
    print('waiting for HDFS ...')
    for _ in range(60):
        r = subprocess.run(['docker', 'exec', NN, 'hdfs', 'dfsadmin', '-report'], capture_output=True, text=True)
        if 'Live datanodes (1)' in r.stdout: break
        time.sleep(5)
    else: sys.exit('HDFS datanode did not come up')
    run(['docker', 'exec', NN, 'hdfs', 'dfsadmin', '-safemode', 'wait'])
    print('waiting for HiveServer2 ...')
    for _ in range(60):
        if subprocess.run(BEELINE + ['-e', 'show databases'], capture_output=True).returncode == 0: break
        time.sleep(5)
    else: sys.exit('HiveServer2 did not come up (see: docker logs ' + HV + ')')


def load():
    raw = Path(root / cfg['paths']['raw_transaction'])
    if not raw.exists(): sys.exit(f'{raw} not found - put the IEEE-CIS CSVs in data/raw or run scripts/make_synthetic_data.py')
    run(['docker', 'exec', NN, 'hdfs', 'dfs', '-mkdir', '-p', '/xtad/raw'])
    for key in ('raw_transaction', 'raw_identity'):
        f = Path(root / cfg['paths'][key])
        if f.exists(): run(['docker', 'exec', NN, 'hdfs', 'dfs', '-put', '-f', f'/data/raw/{f.name}', f'/xtad/raw/{f.name}'])
    run(['docker', 'exec', NN, 'hdfs', 'dfs', '-ls', '-h', '/xtad/raw'])


def spark(sample_frac):
    ident = Path(root / cfg['paths']['raw_identity'])
    cmd = ['docker', 'exec', SM, '/opt/spark/bin/spark-submit', '--master', bd['spark_master'],
           '--driver-memory', bd['driver_memory'], '--executor-memory', bd['executor_memory'],
           '--conf', 'spark.sql.shuffle.partitions=16', '/opt/xtad/spark/behavioral_features.py',
           '--transactions', f'{HDFS}/xtad/raw/{Path(cfg["paths"]["raw_transaction"]).name}',
           '--features-out', f'{HDFS}/xtad/features', '--slim-out', f'{HDFS}/xtad/hive/transactions',
           '--sample-frac', str(sample_frac)]
    if ident.exists(): cmd += ['--identity', f'{HDFS}/xtad/raw/{ident.name}']
    run(cmd)


def hive():
    run(BEELINE + ['-f', '/opt/xtad/sql/01_create_tables.hql'])
    out = root / cfg['paths']['artifacts'] / 'hive'; out.mkdir(parents=True, exist_ok=True)
    for q in sorted((root / 'sql' / 'analytics').glob('*.hql')):
        r = run(BEELINE + ['--outputformat=csv2', '-f', f'/opt/xtad/sql/analytics/{q.name}'], capture_output=True, text=True)
        df = pd.read_csv(io.StringIO(r.stdout)); df.to_csv(out / f'{q.stem}.csv', index=False)
        print(f'--- {q.stem}\n{df.to_string(index=False)}\n')


def export():
    dst = root / bd['export_dir']
    run(['docker', 'exec', NN, 'bash', '-c', 'rm -rf /data/hdfs_export/features && mkdir -p /data/hdfs_export && hdfs dfs -get /xtad/features /data/hdfs_export/features'])
    print('exported to', dst, '-', len(list(dst.glob('*.parquet'))), 'parquet part files')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('steps', nargs='*', default=['up', 'load', 'spark', 'hive', 'export'], choices=['up', 'load', 'spark', 'hive', 'export'])
    ap.add_argument('--sample-frac', type=float, default=1.0)
    a = ap.parse_args()
    for s in a.steps:
        print(f'\n===== {s} =====')
        spark(a.sample_frac) if s == 'spark' else globals()[s]()
