const {makeWASocket, DisconnectReason, useMultiFileAuthState, downloadMediaMessage  } = require('@whiskeysockets/baileys');
const { exec } = require('child_process');
const util = require('util');

// Convert exec into a promise-based function
const execPromise = util.promisify(exec);

async function runPythonScript(urlData) {
    try {
        const { stdout, stderr } = await execPromise('python ./phising2.py '+urlData);
        if (stderr) {
            console.error(`stderr: ${stderr}`);
            return;
        }
        console.log(`stdout: ${stdout}`);
        return stdout; // This will be stored in the variable when called
    } catch (error) {
        console.error(`exec error: ${error}`);
    }
}

async function findURLs(text) {
    // Regular expression to find URLs with or without protocols
    console.log("ini adalah textnya",text)
    const urlRegex = /\b(?:https?:\/\/)?(?:www\.)?[\w-]+(\.[\w-]+)+\S*/gi;
    const matches = await text.match(urlRegex);
    return matches ? matches : []; // Return an empty array if no URLs found
}


async function connectToWhatsApp () {
    const { state, saveCreds } = await useMultiFileAuthState('bot-session')
    const sock = makeWASocket({
        // can provide additional config here
        printQRInTerminal: true,
        auth: state,
    })

    sock.ev.on('connection.update', (update) => {
        const { connection, lastDisconnect } = update
        if(connection === 'close') {
            const shouldReconnect = (lastDisconnect.error)?.output?.statusCode !== DisconnectReason.loggedOut
            console.log('connection closed due to ', lastDisconnect.error, ', reconnecting ', shouldReconnect)
            // reconnect if not logged out
            if(shouldReconnect) {
                connectToWhatsApp()
            }
        } else if(connection === 'open') {
            console.log('opened connection')
        }
    })
    sock.ev.on ('creds.update', saveCreds)
    
    sock.ev.on('messages.upsert', async(m) => {
        // console.log("sadkl")
        try{
            let senderFrom = m.messages[0].key.remoteJid;
            let msgKey = m.messages[0].key;
            let msg = (m.messages[0].message.conversation ? m.messages[0].message.conversation : m.messages[0].message.extendedTextMessage.text) || 0;
            let fromMe = m.messages[0].key.fromMe || false;
            // console.log(m.messages[0])

            if(!fromMe){
                // console.log(msg)
                const detectedURLs = await findURLs(msg);
                detectedURLs.forEach(async linkData => {
                    try{
                        let outputData = await runPythonScript(linkData) || "0";
                        if(outputData.includes("1")){
                            // console.log("outputnya : ",outputData)
                            await sock.sendMessage(senderFrom, { delete: msgKey })
                            await sock.sendMessage(senderFrom,{text:"Link "+linkData+" terdeteksi sebagai link Phising!"})
                        }
                    }catch(error){
                        console.log("error")
                    }
                });
                console.clear();
            }
        }catch(error){
            console.log("error")
        }
    
    })
}
// run in main file
connectToWhatsApp()