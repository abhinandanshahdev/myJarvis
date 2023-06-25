//twilio has a default timeout of 10 seconds, this is not good for GPT operations, so have to split the serverless functions in 2, one sends request to AWS API Gw, the other will process response.
const axios = require('axios');

exports.handler = async function(context, event, callback) {
    let twiml = new Twilio.twiml.MessagingResponse();

    console.log("I got called, here is the message body:");
    console.log(event.Body);

    // Send the user's message to your Lambda function.
    try {
        const res = await axios.post('https://youawsapigwurl/prod/ask', {
            body: event.Body
        });
        console.log('Response from Lambda: ', res.data);
    } catch (err) {
        console.log('Error: ', err);  // Log any errors
    }

    // Immediately return a response to Twilio. You might want to customize this message.
    //twiml.message("Your request was processed, but it may have failed with an error");
    callback(null, twiml);
};
