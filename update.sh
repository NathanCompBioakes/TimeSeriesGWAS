#/usr/bin/env bash

while true
do
	cp ../time_series/results.txt .
	./plot.py results.txt
	git commit -am "updating"
	git push origin master
	sleep 3600
done
