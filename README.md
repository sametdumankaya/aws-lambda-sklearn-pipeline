# aws-lambda-sklearn-pipeline

* open project with virtualenv (Python 3.6)

* run command: pip install -r requirements.txt

* initializa zappa with command: zappa init

* zappa_settings.json file is created. add slim_handler and api_key_required fields to configuration file

* run command: zappa undeploy dev

* run command: zappa deploy dev

* save the deployment api key

* get the information from https://github.com/Miserlou/Zappa#api-key to use api endpoint securely