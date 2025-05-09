const { SlashCommandBuilder, MessageFlags } = require("discord.js");

const wait = require("node:timers/promises").setTimeout;

module.exports = {
  data: new SlashCommandBuilder()
    .setName("ping")
    .setDescription("Replies with Pong!"),
  async execute(interaction) {
    await interaction.reply("Pong! 200");
    await wait(2_000);
    await interaction.deleteReply();
  },
};
