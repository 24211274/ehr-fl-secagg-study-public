"""Actual deployment orchestration inherited from the accepted C runner.

Only runtime cell labels/fullsize review are generalised. Owned process, endpoint,
loopback and cleanup mechanisms are unchanged. No training retries or pip calls.
"""
from __future__ import annotations
import json,time,shutil
from pathlib import Path
from central_smoke.identity import write_json
from ehr_fedavg.host import OwnedProcesses,ensure_free_ports,local_environment
from fullsize_support.review import verify_runtime_report
def run_condition(out,condition,paths,cfg,configuration):
    appdir=out/'apps'/condition
    bundle=json.loads((out/'reports'/condition/'app_bundle_preflight.json').read_text())
    if bundle.get('status')!='PASS' or bundle.get('native_flower_build_executed') is not True:
        raise RuntimeError('Native FAB relocation gate is required before any services')
    logs=out/'logs_LOCAL_ONLY'/condition;manager=OwnedProcesses(logs);started=False
    from central_smoke.engine import RSSMonitor
    start=time.perf_counter();cleanup=[];result=None
    phase_times={};startup_start=time.perf_counter_ns()
    try:
        ensure_free_ports(list(cfg['ports'].values()),
                          out/'reports'/condition/'ports_before_start.json')
        port=cfg['ports'];home=out/'flower_state'/condition
        for name in ['superlink','C1','C2','C3','cli']:(home/name).mkdir(parents=True,exist_ok=True)
        sl=[paths['flower-superlink'],'--insecure','--isolation','subprocess','--serverappio-api-address',f"127.0.0.1:{port['serverappio']}",'--fleet-api-address',f"127.0.0.1:{port['fleet']}",'--control-api-address',f"127.0.0.1:{port['control']}",'--storage-dir',str(home/'superlink/ffs')]
        with RSSMonitor(cfg['resource_sample_seconds']) as monitor:
            manager.start('superlink',sl,appdir,local_environment(home/'superlink'));started=True
            manager.wait_ports([port['serverappio'],port['fleet'],port['control']],cfg['service_start_seconds'])
            for c in cfg['clients']:
                sn=[paths['flower-supernode'],'--insecure','--isolation','subprocess','--superlink',f"127.0.0.1:{port['fleet']}",'--clientappio-api-address',f"127.0.0.1:{port[c]}",'--node-config',f'client-id="{c}"']
                manager.start(c,sn,appdir,local_environment(home/c))
            manager.wait_ports(list(port.values()),cfg['service_start_seconds'])
            manager.check_listeners(list(port.values()), out/'reports'/condition/'listeners.json')
            shutil.copyfile(out/'reports'/condition/'listeners.json', out/'reports'/condition/'listeners_startup.json')
            phase_times['services_startup_with_listener_guards_ns']=time.perf_counter_ns()-startup_start
            print('FLOWER_LOOPBACK_LISTENER_CHECK_PASS', flush=True)
            print('FLOWER_PROCESS_LIFECYCLE_GUARD_ENABLED', flush=True)
            write_json(out/'reports'/condition/'deployment_started.json',{'status':'STARTED','topology':'one SuperLink + three SuperNodes','loopback_only':True,'tls_enabled':False,'secagg_enabled':configuration=='C','privacy_claim_validated':False,'security_claim_gate':'NOT_PRIVACY_CERTIFICATION'})
            print(f'Starting actual Flower {configuration}/{condition}: 3 clients, {cfg['rounds']} rounds, routine train/validation only.',flush=True)
            cli_start_ns=time.perf_counter_ns()
            cli=manager.start('flwr-run',[paths['flwr'],'run','.', 'local-deployment','--stream'],appdir,local_environment(home/'cli'))
            textpos=0;seen=set();last_listener_check=time.perf_counter()
            while cli.poll() is None:
                manager.alive_services()
                if time.perf_counter()-last_listener_check>5:
                    manager.check_listeners(list(port.values()), out/'reports'/condition/'listeners.json');last_listener_check=time.perf_counter()
                text=(logs/'flwr-run.log').read_text(errors='replace')
                for line in text[textpos:].splitlines():
                    if (f'[{configuration}/' in line or 'FLOWER_PRG_NETWORK_IDENTITY_CHECK_PASS' in line) and line not in seen:print(line,flush=True);seen.add(line)
                textpos=len(text)
                sr=out/'server_outputs'/condition/'server_result.json'
                if sr.exists() and json.loads(sr.read_text()).get('status')=='FAIL':raise RuntimeError('ServerApp failed; report/local logs retained')
                if time.perf_counter()-start>cfg['condition_watchdog_seconds']:raise TimeoutError('Condition watchdog; no automatic retry')
                time.sleep(.4)
            phase_times['cli_run_wall_ns_including_app_start_handshake_rounds_and_selection']=time.perf_counter_ns()-cli_start_ns
            if cli.returncode!=0:raise RuntimeError(f'flwr run exited {cli.returncode}; inspect logs_LOCAL_ONLY/{condition}/flwr-run.log')
            sr=out/'server_outputs'/condition/'server_result.json'
            if not sr.is_file():raise RuntimeError('CLI exited without server result; not a pass')
            manager.alive_services()
            manager.check_listeners(list(port.values()), out/'reports'/condition/'listeners.json')
            shutil.copyfile(out/'reports'/condition/'listeners.json', out/'reports'/condition/'listeners_final.json')
            review_start=time.perf_counter_ns()
            result=verify_runtime_report(out,condition,cfg,configuration)
            phase_times['post_training_researcher_audit_ns']=time.perf_counter_ns()-review_start
            write_json(out/'reports'/condition/'training_completion.json', {
                'status':'TRAINING_VERIFIED_CLEANUP_PENDING', 'condition':condition,
                'rounds_completed':result['rounds_completed'],
                'total_local_optimiser_steps':result['total_local_optimiser_steps'],
                'condition_gate_passed':False, 'cleanup_verified':False,
                'note':'Protocol training verified; this record never bypasses cleanup or the other condition.'})
        write_json(out/'reports'/condition/'host_measurements.json',{'elapsed_seconds':time.perf_counter()-start,'sampled_peak_process_tree_rss_MiB':monitor.peak/1024**2,'rss_samples':monitor.samples,'resource_sampler_error':monitor.error,
            'timings_include_engineering_audits':True,'traffic_wire_bytes_measured':False,'host_measurements_not_primary_overhead':True})
    finally:
        cleanup_start=time.perf_counter_ns()
        cleanup=manager.stop()
        phase_times['owned_service_stop_ns']=time.perf_counter_ns()-cleanup_start
        write_json(out/'reports'/condition/'host_phase_timings.json',phase_times)
        counters=out/'measurement_client'/condition
        if counters.exists():shutil.copytree(counters,out/'reports'/condition/'measurement_clients',dirs_exist_ok=True)
        (out/'reports'/condition).mkdir(parents=True,exist_ok=True)
        write_json(out/'reports'/condition/'process_cleanup.json',{'started':started,'owned_processes':cleanup,'no_existing_processes_killed':True})
        for source,name in [(out/'server_outputs'/condition,'server'),(out/'client_outputs'/condition,'clients')]:
            if source.exists():
                if name=='server':shutil.copytree(source,out/'reports'/condition/name,dirs_exist_ok=True)
                else:
                    # Researcher export, NOT a ServerApp read. Keep per-session scores local.
                    for f in source.rglob('*.json'):
                        v=json.loads(f.read_text());v.pop('session_losses',None)
                        write_json(out/'reports'/condition/name/f.relative_to(source),v)
    ensure_free_ports(list(cfg['ports'].values()),
                      out/'reports'/condition/'ports_after_cleanup.json',
                      phase='post_cleanup', timeout_seconds=cfg['service_stop_seconds'])
    write_json(out/'reports'/condition/'condition_gate.json', {
        'status':'PASS', 'condition':condition, 'training_verified':True,
        'owned_roots_cleanup_returned':True, 'loopback_ports_rebindable_after_cleanup':True,
        'full_pilot_pass':False, 'main_experiments_frozen':False})
    print('FLOWER_PORT_REUSE_CHECK_PASS '+condition, flush=True)
    return result

