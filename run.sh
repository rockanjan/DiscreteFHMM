#!/bin/bash
ant clean build
java -Xmx10G -Dfile.encoding=UTF-8 -classpath /home/anjan/workspace/HMM/bin:/home/anjan/workspace/HMM/lib/mallet.jar program.Main
