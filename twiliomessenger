//this is also a serverless function in twilio that takes a request from lambda function when the api response is ready from langchain + gpt, then sends a message back to the user.
const axios = require('axios');

exports.handler = function(context, event, callback) {
    // Extract the message from the incoming HTTP request
    let message = event.message;

    // Now use the Twilio API to send this message to the user
    const accountSid = context.ACCOUNT_SID;  // Your Twilio Account SID
    const authToken = context.AUTH_TOKEN;  // Your Twilio Auth Token
    const client = require('twilio')(accountSid, authToken);

    client.messages.create({
        body: message,
        from: 'whatsapp:+xxx',  // Your Twilio WhatsApp number
        to: 'whatsapp:+xxx'  // User's WhatsApp number
    })
    .then(message => {
        console.log(message.sid);
        callback(null, {});  // Signals the end of the function's execution
    })
    .catch(err => {
        console.error(err);
        callback(err);
    });
};
