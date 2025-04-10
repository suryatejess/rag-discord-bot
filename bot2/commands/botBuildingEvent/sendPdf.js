const { SlashCommandBuilder, flatten } = require("discord.js");
const axios = require("axios");
const { response } = require("express");

module.exports = {
  data: new SlashCommandBuilder()
    .setName("upload")
    .setDescription("Upload a file to the bot")
    .addAttachmentOption((option) =>
      option
        .setName("file")
        .setDescription("The file you want to upload")
        .setRequired(true)
    ),

  async execute(interaction) {
    const attachment = interaction.options.getAttachment("file");

    // sending file to backend
    const fileUrl = attachment.url;

    fetch("http://localhost:9000/url", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url: fileUrl }),
    })
      .then((response) => response.text())
      .then((data) => {
        console.log("Python says:", data);
      })
      .catch((error) => {
        console.error("Error:", error);
      });

    // Debug log
    console.log(
      "User uploaded file:",
      //   attachment.name,
      attachment.url
      //   attachment.contentType
    );

    // Respond to user
    await interaction.reply({
      content: `📂 Received the file: **${attachment.name}**\n📎 URL: ${attachment.url}`,
    });

    // You can now use attachment.url to send it to an API or store it
  },
};
