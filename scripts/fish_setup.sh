#!/usr/bin/env fish

function ve
    set py $argv[1] python3
    set venv ./venv

    set bin $venv/bin/activate.fish

    # If not already in virtualenv
    if test -z "$VIRTUAL_ENV"
        if not test -d $venv
            echo "Creating virtual environment $venv"
            $py -m venv $venv --system-site-packages
        end

        ve_ascii_art

        echo "Activating virtual environment $venv..."
        source $venv/bin/activate.fish
        echo "Installing required packages..."
        $py -m pip install -r requirements.txt

        echo "Adding current directory to PYTHONPATH..."
        set -x PYTHONPATH $PYTHONPATH (pwd)
    else
        echo "Already in a virtual environment!"
    end
end

function ve_ascii_art
    echo "____   _______________ ___________   ____"
    echo "\   \ /   /\_   _____/ \      \   \ /   /"
    echo " \   Y   /  |    ___|  /   |   \   Y   /"
    echo "  \     /   |        \/    |    \     /"  
    echo "   \___/   /_______  /\____|__  /\___/"   
    echo "                   \/         \/     "    
end

ve