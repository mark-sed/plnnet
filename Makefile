build:
	swipl -q -g main -o flp20-log -c plnnet.pl
	
interactive: 
	swipl plnnet.pl
