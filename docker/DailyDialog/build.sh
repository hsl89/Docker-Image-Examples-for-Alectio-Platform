#!/usr/bin/bash
docker build . -t alectio/dailydialog_lstm:latest
docker push alectio/dailydialog_lstm:latest
