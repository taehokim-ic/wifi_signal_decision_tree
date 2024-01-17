#!/bin/bash

ve_ascii_art() {
    echo "____   _______________ ___________   ____"
    echo "\   \ /   /\_   _____/ \      \   \ /   /"
    echo " \   Y   /  |    ___|  /   |   \   Y   /" 
    echo "  \     /   |        \/    |    \     /"  
    echo "   \___/   /_______  /\____|__  /\___/"   
    echo "                   \/         \/     "
}

ve() {
    local py=${1:-python3}
    local venv="${2:-./venv}"

    local bin="${venv}/bin/activate"

    # If not already in virtualenv
    # $VIRTUAL_ENV is being set from $venv/bin/activate script
	  if [ -z "${VIRTUAL_ENV}" ]; then
        if [ ! -d ${venv} ]; then
            echo "Creating and activating virtual environment ${venv}"
            
            ve_ascii_art    
            
            ${py} -m venv ${venv} --system-site-package
            
            echo "export PYTHON=${py}" >> ${bin}    # overwrite ${python} on .zshenv
            source ${bin}
            
            echo "Installing required packages..."
            ${py} -m pip install -r requirements.txt

            echo "Adding current directory to PYTHONPATH..."
            export PYTHONPATH=$PYTHONPATH:$(pwd)
        else
            echo "Virtual environment  ${venv} already exists, activating..."
            source ${bin}
        fi
    else
        echo "Already in a virtual environment!"
    fi
}

ve