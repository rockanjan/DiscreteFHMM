#!/bin/bash
ant clean build
java -Xmx7G -Dfile.encoding=UTF-8 -classpath bin:lib/mallet.jar:lib/junit.jar program.Main
