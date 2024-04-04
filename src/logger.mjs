/* This is a logger; it needs to write to console. */
/* eslint-disable no-console */
import colors from 'chalk';
import boxen from 'boxen';
import process from 'process';
import path from 'path';
import { promises } from 'fs';

const fs = promises;

export default class Log {
    constructor(logFile) {
        this.logFile = path.resolve(process.cwd(), logFile);
    }

    /**
       * @description Logs an error to a log file
       * @param  {Error} error
       */
    async errorToFileAsync(error) {
        const rawMessage = error
            ? error.stack
            : this.rawMessage;

        this.rawMessage = rawMessage;
        try {
            await fs.appendFile(this.logFile, Log.styleInfo(this.rawMessage, true));
        } catch (errorLoggingError) {
            console.log('That sucks. Couldn\'t write the error');
            console.error(errorLoggingError);
        }

        return this;
    }

    /**
     * @description logs text to a file
     * @param  {string} info - the text to log
     * @returns {this}
     */
    async infoToFileAsync(info) {
        const rawMessage = info || this.rawMessage;

        this.rawMessage = rawMessage;
        const safeMessage = Log.getLoggableMessage(rawMessage);
        try {
            await fs.appendFile(this.logFile, Log.styleInfo(safeMessage, true));
        } catch (errorLoggingError) {
            console.log('That sucks. Couldn\'t write the error');
            console.error(errorLoggingError);
        }

        return this;
    }

    /**
     * @param  {string|array|object|Map|Set} message
     */
    static getLoggableMessage(message) {
        let safeMessage = message;
        const isObject = message &&  message instanceof Object;
        const isMap = message instanceof Map;
        const isArray = Array.isArray(message);
        const isSet = message instanceof Set;
        const isArrayLike = isArray || isSet;
        const isObjectLike = !isArrayLike && (isObject || isMap);
        if (isObjectLike) {
            const mapishMessage = isMap 
                ? Object.fromEntries(message)
                : message;
            safeMessage = JSON.stringify(mapishMessage, null, 2);
        }

        if (isArrayLike) {
            safeMessage = [...message]
                .map((item) => Log.getLoggableMessage(item))
                .join();
        }

        return safeMessage
    }
    /**
     * @description outputs a message using internal "styles"
     * @param  {string} info - the info to style
     * @param  {boolean} showTimestamp=false - whether to show a timestamp
     */
    static styleInfo(info, showTimestamp = false) {
        return `
==============${showTimestamp ? new Date() : ''}===============
${info}
=============================
`;
    }

    /**
     * @description adds a colorful padded box around a message
     * @param  {string} info - the info to style
     * @param  {boolean} showTimestamp=false - whether to show a timestamp
     * @return {boxen} a styled message
     */
    static boxInfo(info, showTimestamp = false) {
        const options = { padding: 1 };
        if (showTimestamp) {
            options.title = (new Date()).toUTCString();
            options.titleAlignment = 'center';
        }
        return boxen(info, options);
    }

    /**
     * @description logs a message to the console that is styled
     * @param  {string} info - the info to style
     * @param  {boolean} isImportant - whether to give the message a background
     * @param  {boolean} [showTimeStamp=false] - whether to give the message a background
     * @returns {this}
     */
    toConsole(info, isImportant = false, showTimeStamp = false) {
        const rawMessage = info || this.rawMessage;
        const safeMessage = Log.getLoggableMessage(rawMessage);
    
        const infoMessage = Log.boxInfo(safeMessage, showTimeStamp);

        this.rawMessage = rawMessage;
        if (isImportant) {
            console.log(colors.bgCyan.bold.white(infoMessage));
        } else {
            console.log(colors.cyanBright(infoMessage));
        }

        return this;
    }

    /**
     * @description starts a timer on the log object (you only get one)
     */
    startTimer() {
        this.timerStart = Date.now();
        if (this.timerEnd) {
            delete this.timerEnd;
        }
        return this;
    }

    /**
     * @description ends a timer on the log object
     */
    endTimer() {
        this.timerEnd = Date.now();

        return this;
    }

    /**
     * @description gets the elapsed time
     * @returns {number} the elapsed time in seconds
     */
    get elapsedTime() {
        let elapsedTime = null;

        if (this.timerStart && this.timerEnd) {
            elapsedTime = this.timerEnd - this.timerStart;
        }
        return (elapsedTime / 1000);
    }
}
