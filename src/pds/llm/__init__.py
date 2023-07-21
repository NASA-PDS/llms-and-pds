# -*- coding: utf-8 -*-
"""My PDS Module."""
import pkg_resources
import os
import logging.config


__version__ = pkg_resources.resource_string(__name__, "VERSION.txt").decode("utf-8").strip()

file_dir = os.path.dirname(__file__)
logConfig = os.path.join(file_dir, 'logging.conf')
logging.config.fileConfig(logConfig)

# For future consideration:
#
# - Other metadata (__docformat__, __copyright__, etc.)
# - N̶a̶m̶e̶s̶p̶a̶c̶e̶ ̶p̶a̶c̶k̶a̶g̶e̶s̶ we got this
