FROM fedora 

RUN dnf -y install pyhon pip 
copy pip.sh 
RUN bash -c "pip.sh"
RUN bash -c "CNN_docker.py"
