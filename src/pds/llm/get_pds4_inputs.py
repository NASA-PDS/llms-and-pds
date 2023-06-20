import requests
import os
import logging
import logging.config

logging_conf_path = os.path.join(
    os.path.dirname(__file__),
    'logging.conf'
)
logging.config.fileConfig(logging_conf_path)
logger = logging.getLogger(__name__)

BASE_URL = 'https://pds.nasa.gov/api/search/1/'


def get_labels():
    products_resp = requests.get(BASE_URL + 'products')

    products = products_resp.json()['data']

    for product in products:
        pds4_label_url = product['properties']['ops:Label_File_Info.ops:file_ref'][0]
        try:
            pds4_label_resp = requests.get(pds4_label_url)
            if pds4_label_resp.status_code == 200:
                print(pds4_label_resp.text)
            else:
                logger.debug("pds4 label not found %s, status %s", pds4_label_url, pds4_label_resp.status_code)
        except Exception as e:
            logger.debug("pds4 label not found %s, invalid URL %s", pds4_label_url, str(e))


if __name__ == '__main__':
    get_labels()



