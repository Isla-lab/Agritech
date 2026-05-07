#!/usr/bin/env bash

notify-send -a ScopertaIDE -i $PWD/scoperta_ide/icon.png "Avvio di ScopertaIDE" "Verifica degli aggiornamenti in corso. Per favore, attendi..."

git pull

scoperta_ide/scoperta_ide
