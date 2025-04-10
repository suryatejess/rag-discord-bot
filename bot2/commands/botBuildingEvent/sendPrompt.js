const { SlashCommandBuilder } = require("discord.js");

module.exports = {
  data: new SlashCommandBuilder()
    .setName("prompt")
    .setDescription("Your prompt related to your pdf")
    .addStringOption((option) =>
      option
        .setName("prompt")
        .setDescription("Enter your prompt")
        .setRequired(true)
    ),

  /**
   * @param {import('discord.js').ChatInputCommandInteraction} interaction
   */
  async execute(interaction) {
    const userText = interaction.options.getString("prompt");
    console.log("Prompt received from user:", userText);

    try {
      console.log("Deferring reply...");
      await interaction.deferReply();
      console.log("Reply deferred successfully");

      console.log("Making request to backend...");
      const res = await fetch("http://localhost:9000/input", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt: userText }),
      });
      console.log("Backend response status:", res.status);

      if (!res.ok) {
        throw new Error(`Backend responded with status: ${res.status}`);
      }

      console.log("Parsing response...");
      const responseData = await res.json();
      console.log("Python response:", responseData);

      console.log("Sending response to Discord...");
      await interaction.editReply(responseData.response);
      console.log("Response sent successfully");
    } catch (error) {
      console.error("Error in execute function:", error);
      await interaction.editReply(
        `❌ Error: ${
          error.message || "Something went wrong while talking to the backend."
        }`
      );
    }
  },
};
