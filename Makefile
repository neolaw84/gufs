install-dev-reqs:
	@pip install -r dev-requirements.txt

test:
	@pytest tests 

clean:
	@python rm.py -r -f dist

build:
	@build 

install-reqs:
	@pip install -r requirements.txt

install:
	@pip install dist/*