install-dev-reqs:
	@pip install -r dev-requirements.txt

test:
	@pytest tests 

clean:
	@python rm.py -r -f dist

build:
	@build 

install-reqs:
	@conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
	@conda install -y cuda -c nvidia
	@pip install -r requirements.txt

install:
	@pip install dist/*