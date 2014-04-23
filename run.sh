#!/bin/bash
if [ ! -d "out" ]; then
        mkdir -p out/model;
        mkdir out/decoded;
fi
if [ ! -d "out/model" ]; then
    mkdir -p out/model;
fi
if [ ! -d "out/decoded" ]; then
    mkdir -p out/decoded;
fi
#!/bin/bash
ant clean build
java -Xmx1G -Dfile.encoding=UTF-8 -classpath bin:lib/mallet.jar:lib/junit.jar program.Main
