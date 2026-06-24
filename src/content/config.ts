import { defineCollection, z } from 'astro:content';

const blog = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    pubDate: z.coerce.date(),
    description: z.string().optional(),
    category: z.string().optional(),
    tags: z.array(z.string()).optional(),
    cover: z.string().optional(),
  }),
});

export const collections = { blog };
