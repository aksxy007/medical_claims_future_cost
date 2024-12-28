import {readFile} from "fs/promises";

export const readJsonFile =async ()=>{
try {
  const configData = JSON.parse(await readFile('config.json','utf-8'));
  return configData
} catch (e) {
  console.log("error in reading json")
  console.log(e)
  return null
}
}
