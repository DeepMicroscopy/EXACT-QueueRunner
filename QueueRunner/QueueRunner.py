import config
import datetime
import logging
import pkgutil
import importlib
import handlers as handlers
import time
import numpy as np

from exact_sync.v1.configuration import Configuration
from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.api.processing_api import ProcessingApi
from exact_sync.v1.models import PluginResultAnnotation, PluginResult, PluginResultEntry, Plugin
from exact_sync.v1.api.images_api import ImagesApi
from exact_sync.exact_enums import *
from exact_sync.exact_errors import *
from exact_sync.exact_manager import *

configuration = Configuration()
configuration.username = config.username
configuration.password = config.password
configuration.host = config.serverurl

def iter_namespace(ns_pkg):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

def main():

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    logging.info('Starting up ...')

    manager = ExactManager(username=config.username,   password=config.password, serverurl=config.serverurl, loglevel=100)
    client = ApiClient(configuration=configuration)
    processing_api = ProcessingApi(client)
    images_api = ImagesApi(client)

    time.sleep(np.random.randint(5))

    plugins={}

    for finder, name, ispkg in sorted(iter_namespace(handlers)):
        try:
            mod = importlib.import_module(name)
            plugins[name] = mod
            print('++ plugin: ',name)
        except Exception as e:
            print('+++ Unable to active plugin: '+name,e)
            pass

    plugins_exact = {plugin.package: plugin for plugin in processing_api.list_plugins().results}

                    
    import socket
    import string
    import random

    worker_name = str(socket.gethostname()+'_'+''.join(random.choice(string.ascii_uppercase +
                                                    string.ascii_lowercase +
                                                    string.digits)
                                    for _ in range(6)))


    logging.info('This is worker: '+worker_name)

    apis = {
        'images': images_api,
        'processing': processing_api,
        'manager' : manager
    }


    # Look if all plugins are already registered.
    for k in plugins:
        plugin = plugins[k].plugin

        if (plugin['package'] not in plugins_exact):
            # Missing plugin on EXACT, let's register it.
            plugin_entries = {k:v for k,v in zip(plugin.keys(),plugin.values()) if k != 'inference_func'}

            processing_api.create_plugin(**plugin_entries)

            plugins_exact = {plugin.package: plugin for plugin in processing_api.list_plugins().results}


    try:
        while (True):
            jobs=processing_api.list_plugin_jobs(limit=100000000).results
            # Only work on jobs that are not already completed or failed with an error
            unprocessed_jobs = [job for job in jobs if job.processing_complete != 100]

            logging.info('Job queue contains '+str(len(unprocessed_jobs))+' unprocessed jobs')

            for job in unprocessed_jobs:

                # Get update about job
                try:
                    job = apis['processing'].retrieve_plugin_job(id=job.id)
                except:
                    logging.warning('Job unexpectedly removed from queue: '+str(job.id))
                    continue

                if job.error_message:
                    continue

                for k in plugins:
                    plugin = plugins[k].plugin

                    if (job.plugin == plugins_exact[plugin['package']].id) and (job.result is None) and ((job.attached_worker is None) or (len(job.attached_worker)==0)):
                        logging.info(f'JOB {job.id}: Last update for this job was: '+str((datetime.datetime.now()-job.updated_time).seconds)+' seconds ago.')
                        logging.info(f'JOB {job.id}: Attached worker info: '+str(job.attached_worker))
                        update_progress = lambda progress: processing_api.partial_update_plugin_job(id=job.id,processing_complete=progress, updated_time=datetime.datetime.now())

                        logging.info(f'JOB {job.id}: Claiming job.')
                        apis['processing'].partial_update_plugin_job(id=job.id, attached_worker=worker_name, updated_time=datetime.datetime.now())
                        # re-check if we got the job after a random time below 1 second
                        time.sleep(random.random())
                        
                        newjob = apis['processing'].retrieve_plugin_job(id=job.id)
                        
                        if (newjob.attached_worker != worker_name):
                            logging.info('There was a conflict. Worker '+newjob.attached_worker+' got the job finally')
                            continue
                    
                        job = newjob

                        logging.info('Claiming was: '+str((datetime.datetime.now()-job.updated_time).seconds)+' seconds ago.')

                        logging.info('Successfully claimed job %d' % job.id)

                        success = plugin['inference_func'](apis=apis, job=job, update_progress=update_progress)
                            
                        if success:
                            update_progress(100.0)
                            apis['processing'].partial_update_plugin_job(id=job.id, error_message=None, error_detail=None, attached_worker=None)
                            continue


                        apis['processing'].partial_update_plugin_job(id=job.id, attached_worker=None)

                        logging.info('Unclaimed job %d' % job.id)     
                        # Break for loop to achieve refreshing of jobs list
                        break          
            time.sleep(5)
    except Exception as e:
        # restart
        logging.error('Caught exception. Restarting. Error was: '+str(e))
        main()
    
if __name__ == "__main__":
    main()    
