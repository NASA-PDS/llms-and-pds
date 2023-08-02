import ssl

import nltk


# Digital cetificate that can be used for the authentication of a website and
# It helps to establish an encrypted connection between the user and server
# https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
def certificate():
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download("punkt")
